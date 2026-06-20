/* ============================================================
   LLVM Advisor — Overview Dashboard
   ============================================================ */

const OverviewView = {
  async render() {
    const snap = State.get('currentSnapshot');
    const container = h('div', {});
    Shell.renderMain(container);

    if (!snap) {
      container.appendChild(h('div', { class: 'getting-started', style: { maxWidth: '560px', margin: '0 auto', padding: '48px 24px' } },
        h('h2', { style: { fontSize: '20px', fontWeight: '600', marginBottom: '8px' } }, 'Welcome to llvm-advisor'),
        h('p', { style: { color: 'var(--text-secondary)', marginBottom: '32px' } }, 'Analyze your build, understand your code, and find optimization opportunities.'),
        h('div', { class: 'step-list', style: { display: 'flex', flexDirection: 'column', gap: '16px' } },
          h('div', { class: 'step-card', style: { background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 'var(--r)', padding: '20px' } },
            h('div', { style: { display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' } },
              h('span', { style: { width: '28px', height: '28px', borderRadius: '50%', background: 'var(--accent)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '13px', fontWeight: '600', flexShrink: '0' } }, '1'),
              h('div', { style: { fontWeight: '500' } }, 'Capture your build')
            ),
            h('div', { style: { color: 'var(--text-secondary)', fontSize: '13px', lineHeight: '1.5' } },
              'From your project directory (where compile_commands.json lives):'
            ),
            h('pre', { style: { background: 'var(--bg2)', padding: '12px', borderRadius: '6px', fontFamily: 'var(--mono)', fontSize: '12px', marginTop: '8px', overflow: 'auto', color: 'var(--fg)' } }, 'llvm-advisor capture'),
            h('div', { style: { color: 'var(--text-muted)', fontSize: '12px', marginTop: '4px' } },
              'Auto-detects build directory. Use ', h('code', {}, '--build-root'), ' to override.'
            )
          ),
          h('div', { class: 'step-card', style: { background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 'var(--r)', padding: '20px', opacity: '0.6' } },
            h('div', { style: { display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' } },
              h('span', { style: { width: '28px', height: '28px', borderRadius: '50%', background: 'var(--bg3)', color: 'var(--fg2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '13px', fontWeight: '600', flexShrink: '0' } }, '2'),
              h('div', { style: { fontWeight: '500' } }, 'Explore results')
            ),
            h('div', { style: { color: 'var(--text-secondary)', fontSize: '13px', lineHeight: '1.5' } },
              'The dashboard will appear here once you capture a snapshot.'
            )
          )
        ),
        h('div', { style: { marginTop: '24px', textAlign: 'center' } },
          h('button', {
            class: 'retry-btn',
            style: { padding: '10px 20px', fontSize: '13px' },
            onClick: () => Shell.loadData()
          }, 'Refresh')
        )
      ));
      return;
    }

    const skeleton = this.renderSkeleton();
    container.appendChild(skeleton);

    // Load units, capabilities, and summary in parallel for speed
    const [unitsRes, capsRes, summaryRes] = await Promise.all([
      API.units(snap.id),
      API.capabilities(),
      API.snapshotSummary(snap.id).catch(() => ({ ok: false, data: null, error: 'summary unavailable' })),
    ]);

    let units = unitsRes.ok && Array.isArray(unitsRes.data) ? unitsRes.data : [];
    let specs = capsRes.ok && Array.isArray(capsRes.data) ? capsRes.data : [];
    let summary = summaryRes.ok && summaryRes.data ? summaryRes.data : {};

    // If both units and summary failed, show error
    if (!unitsRes.ok && !summaryRes.ok) {
      if (skeleton.parentNode) skeleton.parentNode.removeChild(skeleton);
      container.appendChild(UI.errorCard(
        [summaryRes.error, unitsRes.error].filter(Boolean).join('; ') || 'Failed to load snapshot data',
        () => this.render()
      ));
      return;
    }

    // Query core capabilities only — avoid expensive/unstable capabilities
    const coreCaps = ['llvm.ir.summary', 'llvm.ir.function_stats', 'clang.diag.summary',
                      'llvm.obj.summary', 'llvm.remarks.summary', 'llvm.remarks.detail',
                      'llvm.debug.summary', 'clang.ast.summary',
                      'llvm.lto.summary', 'llvm.lto.function_stats'];
    const registeredIds = new Set(specs.map(s => s.id));
    const dashboardCaps = coreCaps.filter(id => registeredIds.size === 0 || registeredIds.has(id));
    let aggregate = { metrics: {}, rows: [], errors: 0, warnings: 0, remarks: 0, unavailable: 0, families: [] };

    const queryRes = dashboardCaps.length
      ? await API.querySnapshot(snap.id, dashboardCaps)
      : { ok: false, data: [] };
    if (queryRes.ok) {
      const queryUnits = Array.isArray(queryRes.data) ? queryRes.data : [];
      aggregate = CapabilityData.aggregate(queryUnits);
    }

    // Remove skeleton
    if (skeleton.parentNode) skeleton.parentNode.removeChild(skeleton);

    // Derive health from summary or aggregate
    let health = typeof summary.health_score === 'number' ? summary.health_score : null;
    if (health == null && aggregate.rows.length) {
      health = 100;
      if (aggregate.errors > 0) health = Math.min(health, 60);
      health -= Math.min(aggregate.warnings, 25);
      health -= Math.min(aggregate.unavailable, 20);
      health = Math.max(0, health);
    }

    // Render: Health gauge with correct unit count
    this._renderHealthSection(container, health, units.length, snap);

    // Render: Unit distribution charts (language + target)
    if (units.length) this._renderUnitsSection(container, units);

    // Render: Detailed metrics and panels
    this._renderDetailsSection(container, summary, aggregate, specs, snap, units);
  },

  renderSkeleton() {
    const s = h('div', { class: 'dashboard-skeleton', style: { padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' } },
      h('div', { style: { display: 'flex', gap: '16px' } },
        h('div', { style: { width: '120px', height: '120px', borderRadius: '50%', background: 'var(--bg2)', animation: 'shimmer 1.5s infinite' } }),
        h('div', { style: { flex: '1', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' } },
          Array.from({ length: 4 }, () => h('div', { style: { height: '80px', background: 'var(--bg2)', borderRadius: '8px', animation: 'shimmer 1.5s infinite' } }))
        )
      ),
      h('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' } },
        h('div', { style: { height: '200px', background: 'var(--bg2)', borderRadius: '8px', animation: 'shimmer 1.5s infinite' } }),
        h('div', { style: { height: '200px', background: 'var(--bg2)', borderRadius: '8px', animation: 'shimmer 1.5s infinite' } })
      )
    );
    return s;
  },

  _renderHealthSection(container, health, unitCount, snap) {
    const headerRow = h('div', { style: { display: 'flex', gap: '16px', marginBottom: '16px', alignItems: 'stretch' } });
    headerRow.appendChild(this.renderHealthGauge(health, unitCount, snap));

    const snaps = State.get('snapshots') || [];
    if (snaps.length > 1) {
      const trendData = snaps.map(s => ({
        label: (s.id || '').slice(0, 6),
        value: s.health_score ?? 0,
      }));
      const spark = UI.sparkline(trendData, { color: 'var(--teal)', width: 160, height: 48 });
      if (spark) {
        headerRow.appendChild(h('div', { style: { display: 'flex', flexDirection: 'column', justifyContent: 'center', minWidth: '160px' } },
          h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginBottom: '4px' } }, 'Health Trend'),
          spark
        ));
      }
    }

    container.appendChild(headerRow);
  },

  _renderUnitsSection(container, units) {
    // Language distribution from units
    const byLang = {};
    units.forEach(u => {
      const lang = u.language || 'unknown';
      byLang[lang] = (byLang[lang] || 0) + 1;
    });
    const langData = Object.entries(byLang)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([label, amount]) => ({ label, amount }));

    // Target distribution
    const byTarget = {};
    units.forEach(u => {
      const t = (u.target_triple || '').split('-')[0] || 'unknown';
      byTarget[t] = (byTarget[t] || 0) + 1;
    });
    const targetData = Object.entries(byTarget)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([label, amount]) => ({ label, amount }));

    if (langData.length || targetData.length) {
      const grid = h('div', { class: 'overview-grid', style: { marginTop: '18px' } });
      if (langData.length) {
        const donutData = langData.map(d => ({ label: d.label, value: d.amount }));
        const donut = UI.donutChart(donutData);
        if (donut) {
          grid.appendChild(h('div', { class: 'chart-section' },
            h('h3', {}, 'Language Distribution'),
            donut
          ));
        }
      }
      if (targetData.length) {
        const section = h('div', { class: 'chart-section' }, h('h3', {}, 'Target Architecture'));
        const tagWrap = h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '12px' } });
        targetData.forEach(d => {
          tagWrap.appendChild(h('div', {
            style: { display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '8px 14px', background: 'var(--bg2)', borderRadius: 'var(--r)', border: '1px solid var(--border)' }
          },
            h('span', { style: { fontWeight: '500' } }, d.label),
            h('span', { class: 'mono', style: { color: 'var(--accent)', fontSize: '16px', fontWeight: '600' } }, String(d.amount)),
            h('span', { class: 'text-muted', style: { fontSize: '11px' } }, d.amount === 1 ? 'unit' : 'units')
          ));
        });
        section.appendChild(tagWrap);
        grid.appendChild(section);
      }
      container.appendChild(grid);
    }
  },

  _renderDetailsSection(container, summary, aggregate, specs, snap, units) {
    const rawMetrics = Object.keys(summary.metrics || {}).length ? { ...summary.metrics } : { ...aggregate.metrics };
    // Override with correctly-tracked top-level values (avoids double-counting across capabilities)
    const overrides = { instructions: summary.instructions ?? aggregate.instructions,
                        functions: summary.functions ?? aggregate.functions,
                        remarks: summary.remarks ?? aggregate.remarks,
                        warnings: summary.warnings ?? aggregate.warnings,
                        errors: summary.errors ?? aggregate.errors };
    for (const [k, v] of Object.entries(overrides)) {
      if (v != null) rawMetrics[k] = v;
    }
    delete rawMetrics.instruction_count;
    delete rawMetrics.function_count;
    delete rawMetrics.remark_count;
    const metrics = rawMetrics;
    const health = typeof summary.health_score === 'number' ? summary.health_score : null;

    // Update header with metrics
    const headerRow = container.querySelector('div[style*="display: flex"]');
    if (headerRow) {
      const metricCards = CapabilityData.selectMetrics(metrics, 4);
      if (metricCards.length) {
        const cards = h('div', { class: 'metric-cards', style: { flex: '1' } });
        metricCards.forEach(c => cards.appendChild(UI.metric(c.label, c.value)));
        // Remove old metric cards if present
        const old = headerRow.querySelector('.metric-cards');
        if (old) headerRow.removeChild(old);
        headerRow.appendChild(cards);
      }
    }

    // Overview grid: first-glance impact plots.
    const grid = h('div', { class: 'overview-grid', style: { marginTop: '18px' } });
    grid.appendChild(this._renderHotspotsPanel(aggregate.rows, snap.id));
    grid.appendChild(this._renderCodeSizePanel(aggregate.rows, snap.id));
    grid.appendChild(this._renderTopUnitsPanel(aggregate.rows, snap.id));
    grid.appendChild(this._renderRemarkPanel(aggregate.rows, snap.id));
    container.appendChild(grid);

    // Remark type breakdown donut
    const remarkTypeData = this._buildRemarkTypeBreakdown(aggregate.rows);
    if (remarkTypeData.length) {
      const donut = UI.donutChart(remarkTypeData);
      if (donut) {
        container.appendChild(h('div', { class: 'chart-section', style: { marginTop: '18px', maxWidth: '480px' } },
          h('h3', {}, 'Remark Breakdown'),
          donut
        ));
      }
    }

    // Pass breakdown from remarks by_pass data
    const queryUnits = aggregate.rows || [];
    const byPass = {};
    queryUnits.forEach(u => {
      const results = CapabilityData.normalizeResults(u.results || []);
      results.filter(r => r.capability.includes('remarks')).forEach(res => {
        const v = res.value || {};
        if (v.by_pass && typeof v.by_pass === 'object') {
          for (const [pass, count] of Object.entries(v.by_pass)) {
            if (typeof count === 'number') byPass[pass] = (byPass[pass] || 0) + count;
          }
        }
      });
    });
    const passEntries = Object.entries(byPass).sort((a, b) => b[1] - a[1]);
    if (passEntries.length) {
      const passData = passEntries.slice(0, 12).map(([label, amount]) => ({ label, amount }));
      container.appendChild(h('div', { class: 'chart-section', style: { marginTop: '18px' } },
        h('h3', {}, 'Top Optimization Passes'),
        UI.barChart(passData)
      ));
    }

    // Family coverage lives in Settings, not Overview
  },

  _buildRemarkTypeBreakdown(rows) {
    const byType = {};
    (rows || []).forEach(r => {
      const results = CapabilityData.normalizeResults(r.results || []);
      results.filter(r => r.capability.includes('remarks')).forEach(res => {
        const v = res.value || {};
        if (v.by_type && typeof v.by_type === 'object') {
          for (const [type, count] of Object.entries(v.by_type)) {
            if (typeof count === 'number') byType[type] = (byType[type] || 0) + count;
          }
        }
      });
    });
    return Object.entries(byType)
      .filter(([, v]) => v > 0)
      .sort((a, b) => b[1] - a[1])
      .map(([label, value]) => ({ label: label.charAt(0).toUpperCase() + label.slice(1), value }));
  },

  _renderCodeSizePanel(rows, snapId) {
    const section = h('div', { class: 'chart-section' },
      h('h3', {}, 'Code Size Distribution')
    );
    const sizes = (rows || [])
      .map(r => r.instructions || 0)
      .filter(n => n > 0)
      .sort((a, b) => a - b);
    if (!sizes.length) {
      section.appendChild(h('div', { class: 'empty-inline' }, 'No instruction count data available.'));
      return section;
    }
    const buckets = [
      { label: 'Small (<1K)', min: 0, max: 1000 },
      { label: 'Medium (1-10K)', min: 1000, max: 10000 },
      { label: 'Large (10-50K)', min: 10000, max: 50000 },
      { label: 'Huge (50K+)', min: 50000, max: Infinity },
    ];
    const flameData = buckets
      .map(b => ({ label: b.label, value: sizes.filter(s => s >= b.min && s < b.max).length }))
      .filter(d => d.value > 0);
    const flame = UI.flameBars(flameData);
    if (flame) {
      section.appendChild(flame);
      const legend = h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '6px', fontSize: '11px' } });
      const colors = ['#5B8DB8', '#5DB8A8', '#D4A574', '#9DB86E', '#C97DB8', '#9B7DB8', '#D48B9B', '#6EC9C4'];
      flameData.forEach((d, i) => {
        legend.appendChild(h('span', { style: { display: 'flex', alignItems: 'center', gap: '4px' } },
          h('i', { style: { width: '8px', height: '8px', borderRadius: '2px', background: colors[i % colors.length], display: 'inline-block', flexShrink: '0' } }),
          `${d.label}: ${d.value}`
        ));
      });
      section.appendChild(legend);
    } else {
      section.appendChild(h('div', { class: 'empty-inline' }, 'All units in one bucket.'));
    }
    return section;
  },

  renderHealthGauge(score, unitCount, snap) {
    const r = 34, cx = 42, cy = 42, sw = 6;
    const circ = 2 * Math.PI * r;
    const pct = score != null ? Math.max(0, Math.min(100, score)) : 0;
    const filled = (pct / 100) * circ;
    const cls = pct >= 80 ? 'excellent' : pct >= 60 ? 'good' : pct >= 40 ? 'fair' : 'poor';
    const strokeColor = { excellent: 'var(--green)', good: 'var(--teal)', fair: 'var(--orange)', poor: 'var(--red)' }[cls];
    const offset = (circ / 4).toFixed(1);

    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', '84');
    svg.setAttribute('height', '84');
    svg.setAttribute('viewBox', '0 0 84 84');

    const bgCircle = document.createElementNS(svgNS, 'circle');
    bgCircle.setAttribute('cx', cx);
    bgCircle.setAttribute('cy', cy);
    bgCircle.setAttribute('r', r);
    bgCircle.setAttribute('fill', 'none');
    bgCircle.setAttribute('stroke', 'var(--bg3)');
    bgCircle.setAttribute('stroke-width', sw);
    svg.appendChild(bgCircle);

    const fgCircle = document.createElementNS(svgNS, 'circle');
    fgCircle.setAttribute('cx', cx);
    fgCircle.setAttribute('cy', cy);
    fgCircle.setAttribute('r', r);
    fgCircle.setAttribute('fill', 'none');
    fgCircle.setAttribute('stroke', strokeColor);
    fgCircle.setAttribute('stroke-width', sw);
    fgCircle.setAttribute('stroke-dasharray', `${filled.toFixed(1)} ${circ.toFixed(1)}`);
    fgCircle.setAttribute('stroke-dashoffset', offset);
    fgCircle.setAttribute('stroke-linecap', 'round');
    fgCircle.setAttribute('transform', `rotate(-90 ${cx} ${cy})`);
    svg.appendChild(fgCircle);

    return h('div', { class: 'health-gauge' },
      h('div', { class: 'gauge-ring' }, svg),
      h('div', { class: 'gauge-info' },
        h('div', { class: `gauge-score ${cls}` }, score != null ? String(Math.round(score)) : '–'),
        h('div', { class: 'gauge-label' }, 'Health Score'),
        h('div', { class: 'gauge-sub' }, [
          `${unitCount} unit${unitCount !== 1 ? 's' : ''}`,
          snap ? timeAgo(snap.created_unix) : '',
          snap ? (snap.id || '').slice(0, 8) : '',
        ].filter(Boolean).join(' · '))
      )
    );
  },

  _renderTopUnitsPanel(rows, snapId) {
    const section = h('div', { class: 'chart-section' },
      h('h3', {}, 'Top Compilation Units by Size')
    );

    const sorted = [...rows]
      .filter(r => (r.instructions || 0) > 0 || (r.functions || 0) > 0)
      .sort((a, b) => (b.instructions || 0) - (a.instructions || 0))
      .slice(0, 10);

    if (!sorted.length) {
      section.appendChild(h('div', { class: 'empty-inline' }, 'No size data available yet'));
      return section;
    }

    const wrap = h('div', { class: 'top-units-wrap' });
    const table = h('table', { class: 'top-units-table' },
      h('thead', {}, h('tr', {},
        h('th', {}, 'Source'),
        h('th', { style: { textAlign: 'right' } }, 'Instructions'),
        h('th', { style: { textAlign: 'right' } }, 'Functions'),
        h('th', { style: { textAlign: 'right' } }, 'Warnings'),
      ))
    );
    const tbody = h('tbody', {});
    sorted.forEach(r => {
      const path = r.source_path || r.unit_id || '–';
      const parts = path.replace(/\\/g, '/').split('/');
      const file = parts.pop() || path;
      const shortDir = parts.length > 2 ? '…/' + parts.slice(-2).join('/') : (parts.length ? parts.join('/') : '');
      tbody.appendChild(h('tr', {
        style: { cursor: 'pointer' },
        onClick: () => Router.navigate(`/units/${encodeURIComponent(r.unit_id)}?snapshot=${encodeURIComponent(snapId)}`),
      },
        h('td', { class: 'path', title: path },
          shortDir ? h('span', { class: 'dir' }, shortDir + '/') : null,
          h('span', { class: 'file' }, file)
        ),
        h('td', { class: 'num' }, formatNumber(r.instructions)),
        h('td', { class: 'num' }, formatNumber(r.functions)),
        h('td', { class: 'num' }, (r.warnings || 0) > 0 ? String(r.warnings) : '–'),
      ));
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    section.appendChild(wrap);
    return section;
  },

  _renderHotspotsPanel(rows, snapId) {
    const section = h('div', { class: 'chart-section impact-panel' },
      h('h3', {}, 'Most Important Signals')
    );
    const totals = (rows || []).reduce((acc, r) => {
      acc.errors += Number(r.errors || 0);
      acc.warnings += Number(r.warnings || 0);
      acc.remarks += Number(r.remarks || 0);
      acc.unavailable += Number(r.unavailable || 0);
      return acc;
    }, { errors: 0, warnings: 0, remarks: 0, unavailable: 0 });
    const cards = [
      { key: 'errors', label: 'Errors', value: totals.errors, tone: 'danger', route: '/units' },
      { key: 'warnings', label: 'Warnings', value: totals.warnings, tone: 'warn', route: '/units' },
      { key: 'remarks', label: 'Remarks', value: totals.remarks, tone: 'info', route: '/units' },
      { key: 'units', label: 'Units scanned', value: rows.length, tone: 'neutral', route: '/units' },
    ];
    const grid = h('div', { class: 'impact-grid' });
    cards.forEach(card => {
      grid.appendChild(h('button', {
        class: `impact-card ${card.tone}`,
        onClick: () => Router.navigate(card.route),
      },
        h('span', { class: 'impact-value' }, formatNumber(card.value)),
        h('span', { class: 'impact-label' }, card.label)
      ));
    });
    section.appendChild(grid);

    const worst = [...(rows || [])]
      .filter(r => (r.errors || 0) > 0 || (r.warnings || 0) > 0 || (r.remarks || 0) > 0)
      .sort((a, b) => ((b.errors || 0) * 1000 + (b.warnings || 0) * 10 + (b.remarks || 0)) -
                      ((a.errors || 0) * 1000 + (a.warnings || 0) * 10 + (a.remarks || 0)))
      .slice(0, 5);
    if (worst.length) {
      const list = h('div', { class: 'hotspot-list' });
      worst.forEach(r => {
        const path = r.source_path || r.unit_id || '';
        const file = path.replace(/\\/g, '/').split('/').pop() || path;
        list.appendChild(h('button', {
          class: 'hotspot-row',
          title: path,
          onClick: () => Router.navigate(`/units/${encodeURIComponent(r.unit_id)}?snapshot=${encodeURIComponent(snapId)}`),
        },
          h('span', { class: 'hotspot-name' }, file),
          h('span', { class: 'hotspot-counts' }, [
            r.errors ? `${r.errors} err` : '',
            r.warnings ? `${r.warnings} warn` : '',
            r.remarks ? `${r.remarks} rem` : '',
          ].filter(Boolean).join(' · '))
        ));
      });
      section.appendChild(list);
    } else {
      section.appendChild(h('div', { class: 'empty-inline' }, 'No errors, warnings, or remarks reported yet.'));
    }
    return section;
  },

  _renderRemarkPanel(rows, snapId) {
    const section = h('div', { class: 'chart-section' },
      h('h3', {}, 'Remark Distribution by Unit')
    );
    const ranked = [...(rows || [])]
      .filter(r => (r.remarks || 0) > 0)
      .sort((a, b) => (b.remarks || 0) - (a.remarks || 0))
      .slice(0, 8);
    if (!ranked.length) {
      section.appendChild(h('div', { class: 'empty-inline' }, 'No optimization remarks available.'));
      return section;
    }
    const flameItems = ranked.map(r => {
      const path = r.source_path || r.unit_id || '';
      const file = path.replace(/\\/g, '/').split('/').pop() || path;
      return { label: file, value: r.remarks || 0 };
    });
    const flame = UI.flameBars(flameItems);
    if (flame) section.appendChild(flame);
    const legend = h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px', fontSize: '11px' } });
    const colors = ['#5B8DB8', '#5DB8A8', '#D4A574', '#9DB86E', '#C97DB8', '#9B7DB8', '#D48B9B', '#6EC9C4'];
    ranked.forEach((r, i) => {
      const path = r.source_path || r.unit_id || '';
      const file = path.replace(/\\/g, '/').split('/').pop() || path;
      legend.appendChild(h('span', {
        style: { display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' },
        onClick: () => Router.navigate(`/units/${encodeURIComponent(r.unit_id)}?snapshot=${encodeURIComponent(snapId)}`),
      },
        h('i', { style: { width: '8px', height: '8px', borderRadius: '2px', background: colors[i % colors.length], display: 'inline-block', flexShrink: '0' } }),
        `${file}: ${formatNumber(r.remarks)}`
      ));
    });
    section.appendChild(legend);
    return section;
  },
};
