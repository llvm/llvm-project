/* ============================================================
   LLVM Advisor — Settings View
   ============================================================ */

const SettingsView = {
  async render() {
    const container = h('div', {});
    container.appendChild(h('div', { class: 'section-header' }, 'Settings & Diagnostics'));

    const grid = h('div', { class: 'settings-grid' });
    container.appendChild(grid);
    Shell.renderMain(container);

    const [healthState, capsRes, snapSumRes] = await Promise.all([
      Promise.resolve(State.get('health')),
      API.capabilities(),
      (State.get('currentSnapshot')
        ? API.snapshotSummary(State.get('currentSnapshot').id).catch(() => ({ ok: false }))
        : Promise.resolve({ ok: false })),
    ]);

    const specs = Array.isArray(capsRes.data) ? capsRes.data : [];
    const summary = snapSumRes.ok && snapSumRes.data ? snapSumRes.data : {};
    const families = summary.families || [];

    // Left column: Server Info + Snapshots stacked
    const leftCol = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '16px' } });
    leftCol.appendChild(this._renderServerCard(healthState));
    leftCol.appendChild(this._renderSnapshotsCard());
    grid.appendChild(leftCol);

    // Right column: Capability Families
    grid.appendChild(this._renderFamilyCard(families, specs));

    // Full-width: Registered Capabilities
    grid.appendChild(this._renderCapabilitiesCard(specs));
  },

  _renderServerCard(health) {
    const card = h('div', { class: 'settings-card' }, h('h3', {}, 'Server Info'));
    const kvs = [
      ['Status', health?.ok !== false ? 'OK' : 'Error'],
      ['Store', health?.store || '~/.local/share/llvm-advisor'],
      ['Snapshots', String(health?.snapshots ?? 0)],
      ['Units', String(health?.units ?? 0)],
    ];
    kvs.forEach(([k, v]) => {
      card.appendChild(h('div', { class: 'kv' },
        h('span', { class: 'k' }, k),
        h('span', { class: 'v mono', style: k === 'Store' ? { fontSize: '11px', wordBreak: 'break-all' } : {} }, v)
      ));
    });
    const themeLabel = h('span', { class: 'v mono' }, document.documentElement.classList.contains('dark') ? 'Dark' : 'Light');
    card.appendChild(h('div', { class: 'kv', style: { alignItems: 'center' } },
      h('span', { class: 'k' }, 'Theme'),
      h('button', { class: 'dd-trigger', onClick: () => {
        Theme.toggle();
        themeLabel.textContent = Theme.isDark() ? 'Dark' : 'Light';
        Shell.updateThemeIcon();
      } }, themeLabel)
    ));
    return card;
  },

  _renderFamilyCard(families, specs) {
    const card = h('div', { class: 'settings-card' }, h('h3', {}, 'Capability Coverage'));

    if (!families.length && !specs.length) {
      card.appendChild(h('div', { class: 'text-muted', style: { fontSize: '12px' } }, 'No snapshot selected or no data available.'));
      return card;
    }

    // Build family totals from specs
    const familyTotals = {};
    specs.forEach(s => {
      const fam = CapabilityData.category(s.id || '');
      if (!familyTotals[fam]) familyTotals[fam] = { total: 0, available: 0 };
      familyTotals[fam].total++;
    });
    families.forEach(f => {
      if (familyTotals[f.family]) {
        familyTotals[f.family].available = f.available;
      }
    });

    // Radar chart
    const axes = Object.entries(familyTotals)
      .filter(([, d]) => d.total > 0)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([label, d]) => ({ label, value: d.available, max: d.total }));

    if (axes.length >= 3) {
      card.appendChild(UI.radarChart(axes, { size: 220 }));
    }

    // Family cards
    const familyGrid = h('div', { class: 'family-grid', style: { marginTop: '16px' } });
    Object.entries(familyTotals).sort(([a], [b]) => a.localeCompare(b)).forEach(([family, data]) => {
      const pct = data.total > 0 ? Math.round((data.available / data.total) * 100) : 0;
      const cls = pct === 100 ? 'full' : pct > 0 ? 'partial' : 'none';
      familyGrid.appendChild(h('div', { class: 'family-card' },
        h('div', { class: 'family-card-head' },
          h('div', { class: 'family-card-name' }, family),
          h('div', { class: `family-card-pct ${cls}` }, `${pct}%`)
        ),
        h('div', { class: 'family-bar' },
          h('div', { class: `family-bar-fill ${cls}`, style: { width: `${pct}%` } })
        ),
        h('div', { class: 'family-card-sub' },
          h('span', {}, `${data.available} available`),
          h('span', {}, `${data.total} total`)
        )
      ));
    });
    card.appendChild(familyGrid);
    return card;
  },

  _renderCapabilitiesCard(specs) {
    const card = h('div', { class: 'settings-card', style: { gridColumn: '1 / -1' } },
      h('h3', {}, `Registered Capabilities (${specs.length})`)
    );

    if (!specs.length) {
      card.appendChild(h('div', { class: 'text-muted', style: { fontSize: '12px' } }, 'No capabilities registered.'));
      return card;
    }

    const byFamily = {};
    specs.forEach(s => {
      const fam = CapabilityData.category(s.id || '');
      if (!byFamily[fam]) byFamily[fam] = [];
      byFamily[fam].push(s);
    });

    Object.entries(byFamily).sort(([a], [b]) => a.localeCompare(b)).forEach(([family, caps]) => {
      const section = h('div', { class: 'cap-section open' });
      const header = h('div', { class: 'cap-section-header', onClick: () => section.classList.toggle('open') },
        h('span', {}, `${family} (${caps.length})`),
        h('span', { class: 'text-muted', style: { fontSize: '11px' } }, '▾')
      );
      const body = h('div', { class: 'cap-section-body' });

      caps.sort((a, b) => (a.id || '').localeCompare(b.id || '')).forEach(cap => {
        body.appendChild(h('div', { class: 'cap-status-row' },
          h('span', { class: 'cap-status-id' }, friendlyCapabilityName(cap.id) || cap.id),
          h('span', { class: 'cap-status-dot implemented', title: 'Registered' }),
          h('span', { class: 'cap-status-lvl mono' }, cap.id || '')
        ));
      });

      section.appendChild(header);
      section.appendChild(body);
      card.appendChild(section);
    });

    return card;
  },

  _renderSnapshotsCard() {
    const card = h('div', { class: 'settings-card' }, h('h3', {}, 'Snapshots'));
    const snaps = State.get('snapshots') || [];

    if (!snaps.length) {
      card.appendChild(h('div', { class: 'text-muted', style: { fontSize: '12px' } }, 'No snapshots captured yet.'));
      return card;
    }

    snaps.forEach(s => {
      const current = State.get('currentSnapshot');
      const isCurrent = current && current.id === s.id;
      card.appendChild(h('div', {
        class: 'kv', style: { padding: '8px 0', cursor: 'pointer', ...(isCurrent ? { borderLeft: '3px solid var(--accent)', paddingLeft: '8px' } : {}) },
        onClick: () => { State.set('currentSnapshot', s); Router.navigate('/'); }
      },
        h('div', {},
          h('div', { class: 'mono', style: { fontSize: '12px' } }, (s.id || '').slice(0, 12)),
          h('div', { class: 'text-muted', style: { fontSize: '11px' } }, timeAgo(s.created_unix)),
        ),
        h('div', { style: { display: 'flex', gap: '8px', alignItems: 'center' } },
          isCurrent ? h('span', { style: { fontSize: '10px', color: 'var(--accent)', fontWeight: '600' } }, 'ACTIVE') : null,
          h('button', {
            class: 'dd-trigger', style: { fontSize: '11px', padding: '2px 8px' },
            onClick: (e) => {
              e.stopPropagation();
              State.set('currentSnapshot', s);
              Router.navigate('/');
            }
          }, 'View')
        )
      ));
    });

    return card;
  },
};
