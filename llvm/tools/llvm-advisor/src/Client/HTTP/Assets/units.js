/* ============================================================
   LLVM Advisor — Unit Explorer View
   ============================================================ */

const UnitsView = {
  _units: [],
  _filtered: [],
  _sort: 'name',
  _order: 'asc',
  _search: '',
  _groupByDir: false,
  _debounceTimer: null,

  async render() {
    const container = h('div', { style: { display: 'flex', flexDirection: 'column', height: 'calc(100vh - var(--topbar-h) - 48px)' } });

    container.appendChild(this.controls());
    container.appendChild(h('div', { class: 'unit-overview', id: 'unit-overview' }));

    // List container
    const listWrap = h('div', { class: 'unit-list', id: 'unit-list', style: { flex: '1', overflowY: 'auto' } });
    container.appendChild(listWrap);

    // Status bar
    container.appendChild(h('div', { class: 'status-bar', id: 'unit-status' }));

    Shell.renderMain(container);
    await this.loadUnits();
  },

  controls() {
    const searchInput = h('input', { placeholder: 'Search files and languages…', value: this._search });
    searchInput.addEventListener('input', () => {
      clearTimeout(this._debounceTimer);
      this._debounceTimer = setTimeout(() => {
        this._search = searchInput.value;
        this.applyFilters();
      }, 200);
    });

    return h('div', { class: 'unit-controls' },
      h('div', { class: 'search-input' },
        renderIcon(Icons.search),
        searchInput
      ),
      this.sortDropdown(),
      h('button', {
        class: `view-toggle ${this._groupByDir ? 'active' : ''}`,
        onClick: e => {
          this._groupByDir = !this._groupByDir;
          e.currentTarget.classList.toggle('active', this._groupByDir);
          this.renderList();
        }
      }, 'Group')
    );
  },

  sortDropdown() {
    const options = [
      { value: 'name', label: 'Filename' },
      { value: 'warnings', label: 'Warnings' },
      { value: 'instructions', label: 'Instructions ↓' },
      { value: 'symbols', label: 'Symbols ↓' },
    ];
    const dd = h('div', { class: 'dropdown' });
    const trigger = h('button', { class: 'dd-trigger', onClick: e => Shell.toggleDropdown(e) },
      `Sort: ${options.find(o => o.value === this._sort)?.label || 'Name'} ▾`
    );
    const menu = h('div', { class: 'dd-menu' });
    options.forEach(opt => {
      menu.appendChild(h('div', {
        class: 'dd-item' + (this._sort === opt.value ? ' selected' : ''),
        onClick: () => {
          this._sort = opt.value;
          this._order = opt.value === 'name' ? 'asc' : 'desc';
          trigger.textContent = `Sort: ${opt.label} ▾`;
          Shell.closeDropdowns();
          this.applyFilters();
        }
      }, opt.label));
    });
    dd.appendChild(trigger);
    dd.appendChild(menu);
    return dd;
  },

  async loadUnits() {
    const snap = State.get('currentSnapshot');
    if (!snap) {
      this._units = [];
      this.applyFilters();
      return;
    }
    const [res, metrics] = await Promise.all([
      API.units(snap.id),
      API.querySnapshot(snap.id, ['llvm.ir.summary', 'clang.diag.summary', 'llvm.obj.summary', 'llvm.remarks.summary']),
    ]);
    const units = Array.isArray(res.data) ? res.data : [];
    const byId = new Map();
    if (metrics.ok && Array.isArray(metrics.data)) {
      CapabilityData.aggregate(metrics.data).rows.forEach(row => byId.set(row.unit_id, row));
    }
    this._units = units.map(unit => ({ ...unit, ...(byId.get(unit.id) || {}) }));
    this.applyFilters();
  },

  applyFilters() {
    let list = [...this._units];
    // Search filter (fuzzy)
    if (this._search) {
      const q = this._search;
      list = list.map(u => {
        const text = (u.source_path || u.id || '') + ' ' + (u.language || '');
        const match = fuzzyMatch(q, text);
        return { unit: u, match };
      }).filter(x => x.match).sort((a, b) => b.match.score - a.match.score).map(x => x.unit);
    }
    // Sort
    const field = this._sort;
    const dir = this._order === 'asc' ? 1 : -1;
    list.sort((a, b) => {
      const av = field === 'name' ? (a.source_path || a.id || '') : (a[field] || 0);
      const bv = field === 'name' ? (b.source_path || b.id || '') : (b[field] || 0);
      if (typeof av === 'string') return av.localeCompare(bv) * dir;
      return (av - bv) * dir;
    });
    this._filtered = list;
    this.renderSummary();
    this.renderList();
    this.updateStatus();
  },

  renderSummary() {
    const el = document.getElementById('unit-overview');
    if (!el) return;
    clearEl(el);
    const total = this._filtered.length;
    const warnings = this._filtered.reduce((s, u) => s + Number(u.warnings || 0), 0);
    const errors = this._filtered.reduce((s, u) => s + Number(u.errors || 0), 0);
    const remarks = this._filtered.reduce((s, u) => s + Number(u.remarks || 0), 0);
    const instructions = this._filtered.reduce((s, u) => s + Number(u.instructions || 0), 0);
    [
      ['Units', total, 'neutral'],
      ['Errors', errors, errors ? 'danger' : 'neutral'],
      ['Warnings', warnings, warnings ? 'warn' : 'neutral'],
      ['Remarks', remarks, remarks ? 'info' : 'neutral'],
      ['Instructions', instructions, 'neutral'],
    ].forEach(([label, value, tone]) => {
      el.appendChild(h('div', { class: `unit-summary-card ${tone}` },
        h('span', { class: 'unit-summary-value' }, formatNumber(value)),
        h('span', { class: 'unit-summary-label' }, label)
      ));
    });
  },

  renderList() {
    const wrap = document.getElementById('unit-list');
    if (!wrap) return;
    clearEl(wrap);

    if (this._filtered.length === 0) {
      wrap.appendChild(h('div', { class: 'empty-state' },
        h('div', {}, this._units.length === 0 ? 'No units in snapshot' : 'No matching units'),
        h('div', { class: 'reason' }, this._search ? `No results for "${this._search}"` : 'Capture a snapshot to populate units')
      ));
      return;
    }

    if (this._groupByDir) {
      this.renderGrouped(wrap);
    } else {
      this.renderFlat(wrap);
    }
  },

  renderFlat(wrap) {
    // Virtualized: render only visible rows (simple version)
    const frag = document.createDocumentFragment();
    const limit = Math.min(this._filtered.length, 500); // cap for DOM perf
    for (let i = 0; i < limit; i++) {
      frag.appendChild(this.unitRow(this._filtered[i]));
    }
    wrap.appendChild(frag);
  },

  renderGrouped(wrap) {
    const groups = new Map();
    this._filtered.forEach(u => {
      const path = u.source_path || u.id || '';
      const dir = path.includes('/') ? path.substring(0, path.lastIndexOf('/')) : '.';
      if (!groups.has(dir)) groups.set(dir, []);
      groups.get(dir).push(u);
    });

    groups.forEach((units, dir) => {
      const totalInst = units.reduce((s, u) => s + (u.instructions || 0), 0);
      const totalWarn = units.reduce((s, u) => s + (u.warnings || 0), 0);
      const dirRow = h('div', { class: 'dir-row' },
        renderIcon(Icons.folder),
        h('span', {}, `${dir}/`),
        h('span', { class: 'text-muted', style: { marginLeft: 'auto', fontSize: '11px' } },
          `${units.length} units · inst:${formatNumber(totalInst)} · ⚠${totalWarn}`)
      );

      const children = h('div', { style: { display: 'none' } });
      units.forEach(u => children.appendChild(this.unitRow(u)));

      dirRow.addEventListener('click', () => {
        children.style.display = children.style.display === 'none' ? 'block' : 'none';
      });

      wrap.appendChild(dirRow);
      wrap.appendChild(children);
    });
  },

  unitRow(unit) {
    const path = unit.source_path || unit.id || '';
    const lastSlash = path.lastIndexOf('/');
    const dir = lastSlash >= 0 ? path.substring(0, lastSlash + 1) : '';
    const file = lastSlash >= 0 ? path.substring(lastSlash + 1) : path;
    const lang = unit.language || '';
    const target = (unit.target_triple || '').split('-')[0] || '';
    const warns = unit.warnings || 0;
    const errs = unit.errors || 0;
    const inst = unit.instructions;
    const symbols = unit.symbols;

    const row = h('div', { class: 'unit-row', onClick: () => {
      const snap = State.get('currentSnapshot');
      Router.navigate(`/units/${encodeURIComponent(unit.id)}${snap ? '?snapshot=' + encodeURIComponent(snap.id) : ''}`);
    }},
      h('div', { class: 'filepath' },
        h('span', { class: 'dir' }, dir),
        h('span', { class: 'file' }, file)
      ),
      h('div', { class: 'unit-metadata' },
        h('span', { class: 'lang-badge' }, lang || '–'),
        h('span', { class: 'target mono' }, target || '–')
      ),
      h('div', { class: 'unit-signals' },
        errs > 0 ? h('span', { class: 'signal-pill danger' }, `${errs} errors`) : null,
        warns > 0 ? h('span', { class: 'signal-pill warn' }, `${warns} warnings`) : null,
        (unit.remarks || 0) > 0 ? h('span', { class: 'signal-pill info' }, `${unit.remarks} remarks`) : null,
        errs === 0 && warns === 0 && !unit.remarks ? h('span', { class: 'signal-pill neutral' }, 'clean') : null
      ),
      h('div', { class: 'unit-size' },
        h('span', { class: 'mono-val' }, inst != null ? formatNumber(inst) : '–'),
        h('span', { class: 'unit-size-label' }, 'inst')
      ),
      h('div', { class: 'unit-size' },
        h('span', { class: 'mono-val' }, symbols != null ? formatNumber(symbols) : '–'),
        h('span', { class: 'unit-size-label' }, 'sym')
      ),
    );

    return row;
  },

  showDetail(unit) {
    // Hover should not open panels; unit details are opened intentionally by click.
  },

  updateStatus() {
    const bar = document.getElementById('unit-status');
    if (!bar) return;
    clearEl(bar);
    const total = this._units.length;
    const shown = this._filtered.length;
    const errs = this._filtered.filter(u => u.errors > 0).length;
    bar.appendChild(h('span', {}, `Showing ${formatNumber(shown)} units`));
    if (shown !== total) {
      bar.appendChild(h('span', { class: 'sep' }, ' · '));
      bar.appendChild(h('span', {}, `filtered from ${formatNumber(total)}`));
    }
    if (errs > 0) {
      bar.appendChild(h('span', { class: 'sep' }, ' · '));
      bar.appendChild(h('span', { style: { color: 'var(--red)' } }, `${errs} with errors`));
    }
  },
};
