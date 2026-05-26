/* ============================================================
   LLVM Advisor — Timeline View
   ============================================================ */

const TimelineView = {
  _metrics: ['unit_count', 'instruction_count', 'health_score'],
  _colors: {
    unit_count: '#5B8DB8',
    instruction_count: '#5DB8A8',
    health_score: '#6EC9C4',
    warning_count: '#D4A574',
    error_count: '#D48B9B',
  },
  _snapData: [],

  async render() {
    const container = h('div', {});

    const chips = h('div', { class: 'metric-chips' });
    ['unit_count', 'instruction_count', 'health_score', 'warning_count', 'error_count'].forEach(m => {
      const label = m.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      const chip = h('div', {
        class: 'metric-chip' + (this._metrics.includes(m) ? ' active' : ''),
        onClick: () => {
          const idx = this._metrics.indexOf(m);
          if (idx >= 0) this._metrics.splice(idx, 1);
          else if (this._metrics.length < 4) this._metrics.push(m);
          chip.classList.toggle('active');
          this._drawChart();
        },
      },
        h('span', { class: 'chip-dot', style: { background: this._colors[m] || 'var(--text-muted)' } }),
        label
      );
      chips.appendChild(chip);
    });
    container.appendChild(chips);

    container.appendChild(h('div', { class: 'timeline-chart', id: 'timeline-chart-container' }));

    container.appendChild(h('div', { class: 'metric-cards', id: 'timeline-metrics', style: { marginBottom: '18px' } }));

    container.appendChild(h('div', { class: 'section-header' }, 'Snapshots'));
    container.appendChild(h('div', { class: 'snapshot-list', id: 'snapshot-list' }));

    Shell.renderMain(container);
    await this._loadData();
  },

  async _loadData() {
    const snaps = State.get('snapshots') || [];
    if (!snaps.length) {
      this._renderSnapList([]);
      return;
    }

    const summaries = await Promise.all(snaps.map(s => API.snapshotSummary(s.id)));
    this._snapData = snaps.map((s, i) => {
      const sum = summaries[i].ok && summaries[i].data ? summaries[i].data : {};
      return {
        ...s,
        unit_count: sum.unit_count ?? s.unit_count ?? 0,
        instruction_count: sum.instructions ?? (sum.metrics || {}).instruction_count ?? 0,
        health_score: sum.health_score ?? 0,
        warning_count: sum.warnings ?? (sum.metrics || {}).warnings ?? 0,
        error_count: sum.errors ?? (sum.metrics || {}).errors ?? 0,
        remark_count: sum.remarks ?? (sum.metrics || {}).remark_count ?? 0,
        function_count: sum.functions ?? (sum.metrics || {}).function_count ?? 0,
      };
    });

    this._renderSnapList(this._snapData);
    this._renderMetricCards();
    this._drawChart();
  },

  _renderMetricCards() {
    const el = document.getElementById('timeline-metrics');
    if (!el || !this._snapData.length) return;
    clearEl(el);
    const latest = this._snapData[0];
    const metricDefs = [
      { key: 'unit_count', label: 'Units' },
      { key: 'instruction_count', label: 'Instructions' },
      { key: 'health_score', label: 'Health' },
      { key: 'remark_count', label: 'Remarks' },
      { key: 'function_count', label: 'Functions' },
    ];
    metricDefs.forEach(m => {
      const val = latest[m.key] ?? 0;
      let delta = null, deltaCls = 'neutral';
      if (this._snapData.length > 1) {
        const prev = this._snapData[1];
        const d = (latest[m.key] ?? 0) - (prev[m.key] ?? 0);
        if (d !== 0) {
          const sign = d > 0 ? '+' : '';
          const isGood = m.key === 'health_score' ? d > 0 : m.key === 'warning_count' || m.key === 'error_count' ? d < 0 : null;
          deltaCls = isGood === true ? 'improvement' : isGood === false ? 'regression' : 'neutral';
          delta = `${sign}${formatNumber(d)} vs prev`;
        }
      }
      el.appendChild(UI.metric(m.label, val, delta, deltaCls));
    });
  },

  _drawChart() {
    const container = document.getElementById('timeline-chart-container');
    if (!container) return;
    clearEl(container);
    const data = this._snapData;
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.style.width = '100%';
    svg.style.height = '220px';
    container.appendChild(svg);

    if (data.length < 2) {
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', '50%'); text.setAttribute('y', '50%');
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', 'var(--fg3)'); text.setAttribute('font-size', '12');
      text.textContent = data.length === 1 ? 'Add another snapshot to see trends' : 'Capture snapshots to see trends';
      svg.appendChild(text);
      return;
    }

    const w = 800, ht = 220, padL = 48, padR = 16, padT = 20, padB = 36;
    svg.setAttribute('viewBox', `0 0 ${w} ${ht}`);
    const chartW = w - padL - padR;
    const chartH = ht - padT - padB;

    // Horizontal grid lines
    for (let i = 0; i <= 4; i++) {
      const y = padT + (chartH * i / 4);
      const line = document.createElementNS(svgNS, 'line');
      line.setAttribute('x1', padL); line.setAttribute('y1', y);
      line.setAttribute('x2', w - padR); line.setAttribute('y2', y);
      line.setAttribute('stroke', 'rgba(142,142,147,0.12)');
      line.setAttribute('stroke-width', '1');
      svg.appendChild(line);
    }

    const xStep = chartW / (data.length - 1);

    this._metrics.forEach(m => {
      const values = data.map(s => Number(s[m]) || 0);
      const max = Math.max(...values, 1);
      const min = Math.min(...values, 0);
      const range = max - min || 1;
      const color = this._colors[m] || 'var(--accent)';

      const points = data.map((_, i) => {
        const x = padL + i * xStep;
        const y = padT + chartH - ((values[i] - min) / range) * chartH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });

      // Area fill
      const areaPoints = `${padL},${padT + chartH} ${points.join(' ')} ${(padL + (data.length - 1) * xStep).toFixed(1)},${padT + chartH}`;
      const area = document.createElementNS(svgNS, 'polygon');
      area.setAttribute('points', areaPoints);
      area.setAttribute('fill', color);
      area.setAttribute('opacity', '0.08');
      svg.appendChild(area);

      const poly = document.createElementNS(svgNS, 'polyline');
      poly.setAttribute('points', points.join(' '));
      poly.setAttribute('fill', 'none');
      poly.setAttribute('stroke', color);
      poly.setAttribute('stroke-width', '2');
      poly.setAttribute('stroke-linejoin', 'round');
      svg.appendChild(poly);

      data.forEach((_, i) => {
        const [x, y] = points[i].split(',');
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('cx', x); circle.setAttribute('cy', y);
        circle.setAttribute('r', '3.5'); circle.setAttribute('fill', color);
        svg.appendChild(circle);
      });

      // Y-axis labels for first metric only
      if (m === this._metrics[0]) {
        for (let i = 0; i <= 4; i++) {
          const val = min + (range * (4 - i) / 4);
          const y = padT + (chartH * i / 4);
          const text = document.createElementNS(svgNS, 'text');
          text.setAttribute('x', String(padL - 6));
          text.setAttribute('y', String(y + 3));
          text.setAttribute('text-anchor', 'end');
          text.setAttribute('fill', 'var(--fg3)');
          text.setAttribute('font-size', '9');
          text.setAttribute('font-family', 'var(--mono)');
          text.textContent = val >= 1000 ? (val / 1000).toFixed(1) + 'k' : String(Math.round(val));
          svg.appendChild(text);
        }
      }
    });

    // X-axis labels
    data.forEach((s, i) => {
      const x = padL + i * xStep;
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', x); text.setAttribute('y', ht - 8);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', 'var(--fg3)');
      text.setAttribute('font-size', '9');
      text.setAttribute('font-family', 'var(--mono)');
      text.textContent = (s.id || '').slice(0, 6);
      svg.appendChild(text);
    });

    // Legend
    const legendX = w - padR - this._metrics.length * 100;
    this._metrics.forEach((m, i) => {
      const x = legendX + i * 100;
      const rect = document.createElementNS(svgNS, 'rect');
      rect.setAttribute('x', x); rect.setAttribute('y', '4');
      rect.setAttribute('width', '8'); rect.setAttribute('height', '8');
      rect.setAttribute('rx', '2');
      rect.setAttribute('fill', this._colors[m] || 'var(--accent)');
      svg.appendChild(rect);

      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', String(x + 12)); text.setAttribute('y', '12');
      text.setAttribute('fill', 'var(--fg3)');
      text.setAttribute('font-size', '9');
      text.setAttribute('font-family', 'var(--mono)');
      text.textContent = m.replace(/_/g, ' ');
      svg.appendChild(text);
    });
  },

  _renderSnapList(snaps) {
    const el = document.getElementById('snapshot-list');
    if (!el) return;
    clearEl(el);
    if (!snaps.length) {
      el.appendChild(h('div', { class: 'empty-state' }, h('div', {}, 'No snapshots yet')));
      return;
    }
    snaps.forEach((s, idx) => {
      const healthPct = Number(s.health_score) || 0;
      const healthCls = healthPct >= 80 ? 'excellent' : healthPct >= 60 ? 'good' : healthPct >= 40 ? 'fair' : 'poor';
      const healthColors = { excellent: 'var(--green)', good: 'var(--teal)', fair: 'var(--orange)', poor: 'var(--red)' };

      const deltas = h('div', { class: 'snap-row-deltas', style: { display: 'flex', gap: '6px', flexWrap: 'wrap' } });
      if (idx < snaps.length - 1) {
        const prev = snaps[idx + 1];
        const defs = [
          { key: 'instruction_count', label: 'inst' },
          { key: 'health_score', label: 'health' },
          { key: 'unit_count', label: 'units' },
        ];
        defs.forEach(d => {
          const delta = (s[d.key] || 0) - (prev[d.key] || 0);
          if (delta !== 0) {
            const cls = delta > 0 ? 'positive' : 'negative';
            deltas.appendChild(h('span', { class: `snap-delta ${cls}` },
              `${delta > 0 ? '+' : ''}${formatNumber(delta)} ${d.label}`));
          }
        });
      }

      el.appendChild(h('div', { class: 'snap-row', onClick: () => { State.set('currentSnapshot', s); Router.navigate('/'); } },
        h('span', { class: 'snap-id mono' }, (s.id || '').slice(0, 8)),
        h('span', { class: 'snap-date text-secondary' }, timeAgo(s.created_unix)),
        h('span', { class: 'snap-root text-muted mono' }, s.source_root || '–'),
        deltas,
        h('span', { class: 'snap-num mono' }, formatNumber(s.unit_count || 0)),
        h('span', { class: 'snap-health mono', style: { color: healthColors[healthCls] } },
          healthPct > 0 ? String(Math.round(healthPct)) : '–'),
      ));
    });
  },
};

/* ============================================================
   LLVM Advisor — Insights View
   ============================================================ */

const insightEmptyReasons = {
  call_frequency: 'Requires call graph data. Ensure IR function stats are available.',
  header_depth: 'Requires header dependency data. Compile with -H or enable header tracking.',
  diagnostic_delta: 'Requires at least two snapshots to compare diagnostic changes.',
  optimization_delta: 'Requires at least two snapshots to compare optimization remarks.',
  compilation_flow: 'Requires time-trace data. Compile with -ftime-trace.',
  metric_trends: 'Requires IR summary data. Ensure IR bitcode files are available.',
};

const insightNeedsBaseline = new Set(['diagnostic_delta', 'optimization_delta']);

const InsightsView = {
  _running: new Set(),

  async render() {
    this._running = new Set();
    const container = h('div', {});
    container.appendChild(h('div', { class: 'section-header' }, 'Cross-Unit Insights'));
    const grid = h('div', { class: 'insight-grid', id: 'insight-grid' });
    container.appendChild(grid);
    Shell.renderMain(container);

    const snap = State.get('currentSnapshot');
    if (!snap) {
      grid.appendChild(h('div', { class: 'empty-state' }, h('div', {}, 'Select a snapshot first')));
      return;
    }

    const res = await API.insights(snap.id);
    const insights = Array.isArray(res.data) ? res.data : [];

    if (!insights.length) {
      grid.appendChild(h('div', { class: 'empty-state' },
        h('div', {}, 'No insights available'),
        h('div', { class: 'reason' }, res.error || 'No insights registered for this snapshot')));
      return;
    }

    const available = insights.filter(i => i.available);
    const unavailable = insights.filter(i => !i.available);

    if (available.length) {
      available.forEach((insight, idx) => {
        grid.appendChild(this._renderInsightCard(insight, idx, snap.id));
      });
    }

    if (unavailable.length) {
      grid.appendChild(h('div', { class: 'insight-section-label' }, 'Requires Additional Data'));
      unavailable.forEach((insight, idx) => {
        grid.appendChild(this._renderInsightCard(insight, available.length + idx, snap.id));
      });
    }

    available.forEach((insight, idx) => {
      this._runInsight(insight, idx, snap.id);
    });
  },

  _renderInsightCard(insight, idx, snapId) {
    const category = CapabilityData.category(insight.required_capability || '');
    const card = h('div', { class: 'insight-card', id: `insight-card-${idx}` },
      h('div', { class: 'insight-title' }, titleCase(insight.name || 'Unnamed')),
      h('div', { class: 'insight-category text-muted', style: { fontSize: '11px' } }, category),
      h('div', { class: 'insight-desc' }, insight.description || '')
    );

    if (!insight.available) {
      const reason = insightEmptyReasons[insight.name] || insight.reason || 'Additional data sources needed for this analysis.';
      card.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: 'auto', paddingTop: '8px', lineHeight: '1.5' } },
        h('span', { style: { display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: 'var(--fg3)', marginRight: '6px', verticalAlign: 'middle' } }),
        reason
      ));
      return card;
    }

    const body = h('div', { class: 'insight-body', id: `insight-body-${idx}` });
    body.appendChild(h('div', { class: 'insight-skeleton' }));
    card.appendChild(body);
    return card;
  },

  async _runInsight(insight, idx, snapId) {
    if (this._running.has(insight.name)) return;
    this._running.add(insight.name);

    const body = document.getElementById(`insight-body-${idx}`);
    if (!body) return;

    let res = null;
    if (insightNeedsBaseline.has(insight.name)) {
      const snaps = State.get('snapshots') || [];
      const curIdx = snaps.findIndex(s => s.id === snapId);
      for (let i = curIdx + 1; i < snaps.length && !res?.ok; i++) {
        res = await API.insight(snapId, insight.name, snaps[i].id);
      }
      if (!res?.ok) res = await API.insight(snapId, insight.name);
    } else {
      res = await API.insight(snapId, insight.name);
    }
    clearEl(body);

    if (!res.ok) {
      const reason = insightEmptyReasons[insight.name] || 'This insight requires additional capability data that is not yet available.';
      body.appendChild(h('div', { style: { fontSize: '12px', color: 'var(--fg3)', lineHeight: '1.6', padding: '8px 0' } },
        h('span', { style: { display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: 'var(--fg3)', marginRight: '6px', verticalAlign: 'middle' } }),
        reason
      ));
      return;
    }

    const rawData = res.data?.data || res.data;
    if (!rawData || (typeof rawData === 'object' && Object.keys(rawData).length === 0)) {
      body.appendChild(h('div', { class: 'empty-state', style: { minHeight: '80px' } },
        h('div', {}, 'No data to display'),
        h('div', { class: 'reason' }, 'This insight did not find notable patterns in the current snapshot.')));
      return;
    }

    const rendered = this._renderInsightData(insight.name, rawData);
    if (rendered) {
      body.appendChild(rendered);
    } else {
      const normalized = CapabilityData.normalizeResults([
        { capability: insight.required_capability || insight.name, value: rawData }
      ])[0];
      body.appendChild(normalized ? UI.capabilityPanel(normalized) : h('div', { class: 'text-muted', style: { fontSize: '12px' } }, 'No data returned'));
    }
  },

  _renderInsightData(name, data) {
    const d = data || {};
    const wrap = h('div', { class: 'insight-content' });

    if (name === 'pass_impact') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Optimization Hit Rate'), h('strong', { class: 'mono' }, `${(d.optimization_hit_rate_pct || 0).toFixed(1)}%`)));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Total Remarks'), h('strong', { class: 'mono' }, formatNumber(d.total_remarks || 0))));
      const byType = d.by_type || {};
      Object.entries(byType).forEach(([k, v]) => {
        metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, `Type ${titleCase(k)}`), h('strong', { class: 'mono' }, formatNumber(v))));
      });
      wrap.appendChild(metrics);
      if (d.by_type && Object.keys(d.by_type).length > 1) {
        const donutData = Object.entries(d.by_type).filter(([, v]) => v > 0).map(([label, value]) => ({ label: titleCase(label), value }));
        const donut = UI.donutChart(donutData, { size: 100 });
        if (donut) wrap.appendChild(donut);
      }
      const passes = Array.isArray(d.top_passes_by_remarks) ? d.top_passes_by_remarks : [];
      if (passes.length) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'Top Passes By Remarks'));
        wrap.appendChild(UI.dataTable(passes.slice(0, 10), { columns: ['count', 'pass', 'pct_of_total'] }));
      }
      return wrap;
    }

    if (name === 'function_complexity') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Total Functions'), h('strong', { class: 'mono' }, formatNumber(d.total_functions || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Total Instructions'), h('strong', { class: 'mono' }, formatNumber(d.total_instructions || 0))));
      if (d.p90_instruction_threshold) metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'P90 Threshold'), h('strong', { class: 'mono' }, formatNumber(d.p90_instruction_threshold))));
      wrap.appendChild(metrics);
      const fns = (Array.isArray(d.top_by_instructions) ? d.top_by_instructions : []).filter(f => f.name && !isCorruptedString(f.name));
      if (fns.length) {
        const barData = fns.slice(0, 8).map(f => ({ label: f.name, amount: f.instructions || f.basic_blocks || 0 }));
        const chart = UI.barChart(barData);
        if (chart) wrap.appendChild(chart);
      }
      return wrap;
    }

    if (name === 'debug_info') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Debug Info'), h('strong', { class: 'mono' }, d.has_debug_info ? 'Yes' : 'No')));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Coverage'), h('strong', { class: 'mono' }, titleCase(d.coverage || 'unknown'))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Compile Units'), h('strong', { class: 'mono' }, formatNumber(d.compile_units || 0))));
      if (d.max_dwo_version) metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'DWO Version'), h('strong', { class: 'mono' }, String(d.max_dwo_version))));
      wrap.appendChild(metrics);
      const interps = Array.isArray(d.interpretations) ? d.interpretations : [];
      if (interps.length) {
        const list = h('div', { style: { marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' } });
        interps.forEach(msg => {
          list.appendChild(h('div', { style: { fontSize: '12px', color: 'var(--fg2)', lineHeight: '1.5', padding: '6px 10px', background: 'var(--bg2)', borderRadius: 'var(--r)', borderLeft: '3px solid var(--accent)' } }, msg));
        });
        wrap.appendChild(list);
      }
      return wrap;
    }

    if (name === 'section_sizes') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Total Size'), h('strong', { class: 'mono' }, formatBytes(d.total_size || 0))));
      if (d.format && !isCorruptedString(d.format)) metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Format'), h('strong', { class: 'mono' }, d.format)));
      wrap.appendChild(metrics);
      const cats = d.category_breakdown || {};
      const catEntries = Object.entries(cats).filter(([k, v]) => v && v.size > 0 && !isCorruptedString(k)).sort((a, b) => b[1].size - a[1].size);
      if (catEntries.length) {
        const flameItems = catEntries.map(([label, v]) => ({ label: titleCase(label), value: v.size }));
        const flame = UI.flameBars(flameItems);
        if (flame) wrap.appendChild(flame);
        const legend = h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '6px', fontSize: '11px' } });
        const colors = ['#5B8DB8', '#5DB8A8', '#D4A574', '#9DB86E', '#C97DB8', '#9B7DB8', '#D48B9B', '#6EC9C4'];
        catEntries.forEach(([label, v], i) => {
          legend.appendChild(h('span', { style: { display: 'flex', alignItems: 'center', gap: '4px' } },
            h('i', { style: { width: '8px', height: '8px', borderRadius: '2px', background: colors[i % colors.length], display: 'inline-block', flexShrink: '0' } }),
            `${titleCase(label)}: ${formatBytes(v.size)} (${(v.pct_of_total || 0).toFixed(1)}%)`
          ));
        });
        wrap.appendChild(legend);
      }
      const sections = (Array.isArray(d.sections) ? d.sections : []).filter(s => s.name && !isCorruptedString(s.name)).slice(0, 10);
      if (sections.length) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'Top Sections'));
        const barData = sections.map(s => ({ label: s.name, amount: s.size || 0 }));
        wrap.appendChild(UI.barChart(barData));
      }
      return wrap;
    }

    if (name === 'loop_nesting') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Total Loops'), h('strong', { class: 'mono' }, formatNumber(d.total_loops || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Max Depth'), h('strong', { class: 'mono' }, String(d.global_max_depth || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Deep Nesting Threshold'), h('strong', { class: 'mono' }, String(d.deep_nesting_threshold || 3))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Deeply Nested Fns'), h('strong', { class: 'mono' }, formatNumber(d.deeply_nested_functions || 0))));
      wrap.appendChild(metrics);
      const fns = (Array.isArray(d.top_by_nesting) ? d.top_by_nesting : []).filter(f => f.name && !isCorruptedString(f.name));
      if (fns.length) {
        const barData = fns.slice(0, 8).map(f => ({ label: f.name, amount: f.loops || 0 }));
        const chart = UI.barChart(barData);
        if (chart) wrap.appendChild(chart);
      }
      return wrap;
    }

    if (name === 'diagnostic_delta') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Error Delta'), h('strong', { class: 'mono' }, String(d.error_delta || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Warning Delta'), h('strong', { class: 'mono' }, String(d.warning_delta || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Note Delta'), h('strong', { class: 'mono' }, String(d.note_delta || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'New Errors'), h('strong', { class: 'mono' }, String(d.new_errors || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'New Warnings'), h('strong', { class: 'mono' }, String(d.new_warnings || 0))));
      wrap.appendChild(metrics);
      const base = d.baseline || {};
      const prim = d.primary || {};
      if (base.errors != null || prim.errors != null) {
        const items = [
          { label: 'Errors', before: base.errors || 0, after: prim.errors || 0 },
          { label: 'Warnings', before: base.warnings || 0, after: prim.warnings || 0 },
          { label: 'Notes', before: base.notes || 0, after: prim.notes || 0 },
        ];
        const deltaBar = UI.deltaBar(items);
        if (deltaBar) wrap.appendChild(deltaBar);
      }
      const newDiags = Array.isArray(d.new_diagnostics) ? d.new_diagnostics : [];
      if (newDiags.length) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'New Diagnostics'));
        wrap.appendChild(UI.findingList(newDiags.slice(0, 20)));
      } else {
        wrap.appendChild(h('div', { style: { fontSize: '12px', color: 'var(--fg3)', marginTop: '10px' } }, 'No new diagnostics detected between snapshots.'));
      }
      return wrap;
    }

    if (name === 'optimization_delta') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Total Delta'), h('strong', { class: 'mono' }, String(d.total_delta || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Primary Total'), h('strong', { class: 'mono' }, formatNumber(d.primary_total || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Baseline Total'), h('strong', { class: 'mono' }, formatNumber(d.baseline_total || 0))));
      wrap.appendChild(metrics);
      const byType = d.by_type_delta || {};
      const cleanEntries = Object.entries(byType).filter(([k]) => !isCorruptedString(k));
      if (cleanEntries.length) {
        const items = cleanEntries.map(([label, v]) => ({
          label: titleCase(label),
          before: v?.baseline || 0,
          after: v?.primary || 0,
        }));
        const deltaBar = UI.deltaBar(items);
        if (deltaBar) wrap.appendChild(deltaBar);
      }
      const passes = Array.isArray(d.top_changed_passes) ? d.top_changed_passes.filter(p => !isCorruptedString(p.pass || '')) : [];
      if (passes.length) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'Top Changed Passes'));
        wrap.appendChild(UI.dataTable(passes.slice(0, 10)));
      } else {
        wrap.appendChild(h('div', { style: { fontSize: '12px', color: 'var(--fg3)', marginTop: '10px' } }, 'No significant pass-level changes detected between snapshots.'));
      }
      return wrap;
    }

    if (name === 'header_depth') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Max Depth'), h('strong', { class: 'mono' }, String(d.max_depth || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Total Headers'), h('strong', { class: 'mono' }, formatNumber(d.total_headers || 0))));
      wrap.appendChild(metrics);
      const chains = Array.isArray(d.deepest_chains) ? d.deepest_chains : [];
      if (chains.length) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'Deepest Include Chains'));
        wrap.appendChild(UI.dataTable(chains.slice(0, 10)));
      }
      const most = Array.isArray(d.most_included) ? d.most_included : [];
      if (most.length) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'Most Included Headers'));
        wrap.appendChild(UI.dataTable(most.slice(0, 10)));
      }
      if (!chains.length && !most.length && !d.max_depth) {
        wrap.appendChild(h('div', { style: { fontSize: '12px', color: 'var(--fg3)', marginTop: '10px' } }, 'No header dependency data found. Compile with -H to enable.'));
      }
      return wrap;
    }

    if (name === 'call_frequency') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Total Functions'), h('strong', { class: 'mono' }, formatNumber(d.total_functions || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Call Edges'), h('strong', { class: 'mono' }, formatNumber(d.total_call_edges || 0))));
      wrap.appendChild(metrics);
      const fanIn = (Array.isArray(d.top_callers_by_fan_in) ? d.top_callers_by_fan_in : []).filter(f => f.name && !isCorruptedString(f.name));
      const fanOut = (Array.isArray(d.top_callees_by_fan_out) ? d.top_callees_by_fan_out : []).filter(f => f.name && !isCorruptedString(f.name));
      const fanInHasData = fanIn.some(f => (f.incoming_calls || 0) > 0);
      const fanOutHasData = fanOut.some(f => (f.outgoing_calls || 0) > 0);
      if (fanIn.length && fanInHasData) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'Most Called (Fan-In)'));
        const barData = fanIn.slice(0, 8).map(f => ({ label: f.name, amount: f.incoming_calls || 0 }));
        wrap.appendChild(UI.barChart(barData));
      }
      if (fanOut.length && fanOutHasData) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'Highest Fan-Out'));
        const barData = fanOut.slice(0, 8).map(f => ({ label: f.name, amount: f.outgoing_calls || 0 }));
        wrap.appendChild(UI.barChart(barData));
      }
      const hubs = Array.isArray(d.hub_functions) ? d.hub_functions.filter(f => f.name && !isCorruptedString(f.name)) : [];
      if (hubs.length) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'Hub Functions'));
        wrap.appendChild(UI.dataTable(hubs.slice(0, 8), { columns: ['name', 'incoming_calls', 'outgoing_calls'] }));
      }
      if (!fanInHasData && !fanOutHasData && fanOut.length) {
        wrap.appendChild(h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginTop: '12px', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' } }, 'Functions'));
        wrap.appendChild(UI.dataTable(fanOut.slice(0, 10), { columns: ['name', 'outgoing_calls', 'incoming_calls'] }));
      }
      return wrap;
    }

    if (name === 'compilation_flow') {
      const stages = Array.isArray(d.stages) ? d.stages : [];
      const total = d.total_duration_ms || 0;
      const slowest = d.slowest_event || {};
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' },
        h('span', {}, 'Total Time'), h('strong', { class: 'mono' }, `${total} ms`)));
      if (slowest.name) {
        metrics.appendChild(h('div', { class: 'mini-metric' },
          h('span', {}, 'Slowest Event'),
          h('strong', { class: 'mono', style: { fontSize: '10px' } }, slowest.name)));
        metrics.appendChild(h('div', { class: 'mini-metric' },
          h('span', {}, 'Slowest Time'),
          h('strong', { class: 'mono' }, `${Math.round((slowest.duration_us || 0) / 1000)} ms`)));
      }
      wrap.appendChild(metrics);
      if (stages.length) {
        const colors = { frontend: '#5B8DB8', optimizer: '#D4A574', codegen: '#9DB86E', other: '#C97DB8' };
        // Stacked horizontal bar
        const bar = h('div', { style: { display: 'flex', height: '28px', borderRadius: '4px', overflow: 'hidden', margin: '12px 0 4px' } });
        stages.forEach(s => {
          const pct = s.pct_of_total || 0;
          if (pct <= 0) return;
          const color = colors[s.stage] || '#9B7DB8';
          const seg = h('div', {
            style: { width: `${pct}%`, background: color, display: 'flex', alignItems: 'center',
                     justifyContent: 'center', overflow: 'hidden', whiteSpace: 'nowrap' },
            title: `${s.stage}: ${s.duration_ms} ms (${pct}%)`,
          }, pct > 8 ? h('span', { style: { fontSize: '10px', color: '#fff', fontWeight: '600' } }, s.stage) : null);
          bar.appendChild(seg);
        });
        wrap.appendChild(bar);
        // Legend rows
        const legend = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '4px' } });
        stages.forEach(s => {
          const color = colors[s.stage] || '#9B7DB8';
          legend.appendChild(h('div', { style: { display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' } },
            h('i', { style: { width: '10px', height: '10px', borderRadius: '2px', background: color, flexShrink: '0', display: 'inline-block' } }),
            h('span', { style: { color: 'var(--fg2)', minWidth: '80px' } }, s.stage),
            h('span', { class: 'mono' }, `${s.duration_ms} ms`),
            h('span', { style: { color: 'var(--fg3)', marginLeft: '4px' } }, `${s.pct_of_total}%`)
          ));
        });
        wrap.appendChild(legend);
      }
      return wrap;
    }

    if (name === 'metric_trends') {
      const metrics = h('div', { class: 'mini-metrics' });
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Functions'), h('strong', { class: 'mono' }, formatNumber(d.functions || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Instructions'), h('strong', { class: 'mono' }, formatNumber(d.instructions || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Globals'), h('strong', { class: 'mono' }, formatNumber(d.globals || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Instr / Fn'), h('strong', { class: 'mono' }, String(d.instructions_per_function || 0))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Size Class'), h('strong', { class: 'mono' }, titleCase(d.size_class || 'unknown'))));
      metrics.appendChild(h('div', { class: 'mini-metric' }, h('span', {}, 'Density'), h('strong', { class: 'mono' }, titleCase(d.density_class || 'unknown'))));
      wrap.appendChild(metrics);
      if (d.functions > 0 && d.instructions > 0) {
        const donutData = [
          { label: 'Functions', value: d.functions },
          { label: 'Globals', value: d.globals || 0 },
        ].filter(x => x.value > 0);
        if (donutData.length > 1) {
          const donut = UI.donutChart(donutData, { size: 90 });
          if (donut) wrap.appendChild(donut);
        }
      }
      const interps = Array.isArray(d.interpretations) ? d.interpretations : [];
      if (interps.length) {
        const list = h('div', { style: { marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' } });
        interps.forEach(msg => {
          list.appendChild(h('div', { style: { fontSize: '12px', color: 'var(--fg2)', lineHeight: '1.5', padding: '6px 10px', background: 'var(--bg2)', borderRadius: 'var(--r)', borderLeft: '3px solid var(--accent)' } }, msg));
        });
        wrap.appendChild(list);
      }
      return wrap;
    }

    return null;
  },
};

/* ============================================================
   LLVM Advisor — Remarks Explorer View
   ============================================================ */

const RemarksView = {
  async render() {
    const snap = State.get('currentSnapshot');
    const container = h('div', {});
    container.appendChild(h('div', { class: 'section-header' }, 'Optimization Remarks Explorer'));
    Shell.renderMain(container);

    if (!snap) {
      container.appendChild(h('div', { class: 'empty-state' },
        h('div', {}, 'Select a snapshot first')));
      return;
    }

    const skeleton = h('div', { class: 'dashboard-skeleton', style: { padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' } },
      h('div', { style: { height: '80px', background: 'var(--bg2)', borderRadius: '8px', animation: 'shimmer 1.5s infinite' } }),
      h('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' } },
        h('div', { style: { height: '200px', background: 'var(--bg2)', borderRadius: '8px', animation: 'shimmer 1.5s infinite' } }),
        h('div', { style: { height: '200px', background: 'var(--bg2)', borderRadius: '8px', animation: 'shimmer 1.5s infinite' } })
      )
    );
    container.appendChild(skeleton);

    const [relRes, queryRes] = await Promise.all([
      API.get(`/snapshots/${snap.id}/remarks/relational`),
      API.querySnapshot(snap.id, ['llvm.remarks.summary']),
    ]);

    if (skeleton.parentNode) skeleton.parentNode.removeChild(skeleton);

    if (!relRes.ok && !queryRes.ok) {
      container.appendChild(UI.errorCard(
        'No remarks data available. Capture with llvm.remarks.summary enabled.',
        () => this.render()
      ));
      return;
    }

    const rel = relRes.ok && relRes.data ? relRes.data : null;
    const queryUnits = queryRes.ok && Array.isArray(queryRes.data) ? queryRes.data : [];

    // Top-level summary from summary capability
    const byPass = {}, byType = {};
    let totalRemarks = 0;
    queryUnits.forEach(u => {
      const results = CapabilityData.normalizeResults(u.results || []);
      results.filter(r => r.capability === 'llvm.remarks.summary').forEach(res => {
        const v = res.value || {};
        totalRemarks += Number(v.count || v.remark_count || 0);
        if (v.by_pass) Object.entries(v.by_pass).forEach(([p, c]) => { byPass[p] = (byPass[p] || 0) + Number(c); });
        if (v.by_type) Object.entries(v.by_type).forEach(([t, c]) => { byType[t] = (byType[t] || 0) + Number(c); });
      });
    });

    // Header stat row
    const statRow = h('div', { class: 'metric-cards', style: { marginBottom: '18px' } });
    statRow.appendChild(UI.metric('Total Remarks', totalRemarks));
    statRow.appendChild(UI.metric('Units', queryUnits.length));
    if (rel) statRow.appendChild(UI.metric('Relational Rows', rel.count || 0));
    container.appendChild(statRow);

    const grid = h('div', { class: 'overview-grid', style: { marginTop: '0' } });

    // Pass distribution bar chart
    const passEntries = Object.entries(byPass).sort((a, b) => b[1] - a[1]);
    if (passEntries.length) {
      const passData = passEntries.slice(0, 12).map(([label, amount]) => ({ label, amount }));
      const section = h('div', { class: 'chart-section' },
        h('h3', {}, 'Remarks by Pass'),
        UI.barChart(passData)
      );
      grid.appendChild(section);
    }

    // Remark type donut
    const typeEntries = Object.entries(byType).filter(([, v]) => v > 0);
    if (typeEntries.length) {
      const typeData = typeEntries.map(([label, value]) => ({
        label: label.charAt(0).toUpperCase() + label.slice(1),
        value,
      }));
      const donut = UI.donutChart(typeData);
      if (donut) {
        grid.appendChild(h('div', { class: 'chart-section' },
          h('h3', {}, 'By Remark Type'),
          donut
        ));
      }
    }

    // Per-unit remark distribution (long tail)
    const unitRemarks = queryUnits.map(u => {
      const results = CapabilityData.normalizeResults(u.results || []);
      const rem = results.filter(r => r.capability === 'llvm.remarks.summary')
        .reduce((s, r) => s + Number(r.value?.count || r.value?.remark_count || 0), 0);
      return { unit_id: u.unit_id, source_path: u.source_path, remarks: rem };
    }).filter(u => u.remarks > 0).sort((a, b) => b.remarks - a.remarks);

    if (unitRemarks.length) {
      const flameItems = unitRemarks.slice(0, 10).map(u => {
        const path = u.source_path || u.unit_id || '';
        const file = path.replace(/\\/g, '/').split('/').pop() || path;
        return { label: file, value: u.remarks };
      });
      const section = h('div', { class: 'chart-section' },
        h('h3', {}, 'Remarks per Unit (top 10)')
      );
      const flame = UI.flameBars(flameItems);
      if (flame) section.appendChild(flame);
      // Legend with links
      const legend = h('div', { style: { marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '6px', fontSize: '11px' } });
      const colors = ['#5B8DB8', '#5DB8A8', '#D4A574', '#9DB86E', '#C97DB8', '#9B7DB8', '#D48B9B', '#6EC9C4', '#5B8DB8', '#D4A574'];
      unitRemarks.slice(0, 10).forEach((u, i) => {
        const path = u.source_path || u.unit_id || '';
        const file = path.replace(/\\/g, '/').split('/').pop() || path;
        legend.appendChild(h('span', {
          style: { display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' },
          onClick: () => Router.navigate(`/units/${encodeURIComponent(u.unit_id)}?snapshot=${encodeURIComponent(snap.id)}`),
        },
          h('i', { style: { width: '8px', height: '8px', borderRadius: '2px', background: colors[i], display: 'inline-block', flexShrink: '0' } }),
          `${file}: ${formatNumber(u.remarks)}`
        ));
      });
      section.appendChild(legend);
      grid.appendChild(section);
    }

    container.appendChild(grid);

    // Relational table — top (pass, name) tuples
    if (rel && rel.columns && rel.strings) {
      const { columns, strings } = rel;
      const passes = strings.pass || [];
      const names = strings.name || [];
      const types = [null, 'passed', 'missed', 'analysis', 'analysis-fp-commute', 'analysis-aliasing', 'failure'];

      // Count (pass, name) tuples
      const tuples = {};
      const passCols = columns.pass || [];
      const nameCols = columns.name || [];
      const typeCols = columns.type || [];
      for (let i = 0; i < passCols.length; i++) {
        const p = passes[passCols[i]] || '?';
        const n = names[nameCols[i]] || '?';
        const key = `${p}\0${n}`;
        if (!tuples[key]) tuples[key] = { pass: p, name: n, by_type: {} };
        const typeName = types[typeCols[i]] || 'unknown';
        tuples[key].by_type[typeName] = (tuples[key].by_type[typeName] || 0) + 1;
        tuples[key].count = (tuples[key].count || 0) + 1;
      }

      const sorted = Object.values(tuples).sort((a, b) => b.count - a.count).slice(0, 20);
      if (sorted.length) {
        const tableSection = h('div', { class: 'chart-section', style: { marginTop: '18px' } },
          h('h3', {}, `Top (Pass, Remark) Pairs — ${rel.count} total remarks`)
        );
        const table = h('table', { class: 'top-units-table' },
          h('thead', {}, h('tr', {},
            h('th', {}, 'Pass'), h('th', {}, 'Remark'),
            h('th', { style: { textAlign: 'right' } }, 'Count'),
            h('th', { style: { textAlign: 'right' } }, 'Missed'),
            h('th', { style: { textAlign: 'right' } }, 'Passed'),
          ))
        );
        const tbody = h('tbody', {});
        sorted.forEach(t => {
          tbody.appendChild(h('tr', {},
            h('td', { class: 'mono', style: { fontSize: '11px' } }, t.pass),
            h('td', { style: { fontSize: '11px' } }, t.name),
            h('td', { class: 'num' }, formatNumber(t.count)),
            h('td', { class: 'num', style: { color: (t.by_type.missed || 0) > 0 ? 'var(--orange)' : 'var(--fg3)' } },
              t.by_type.missed ? formatNumber(t.by_type.missed) : '–'),
            h('td', { class: 'num', style: { color: (t.by_type.passed || 0) > 0 ? 'var(--green)' : 'var(--fg3)' } },
              t.by_type.passed ? formatNumber(t.by_type.passed) : '–'),
          ));
        });
        table.appendChild(tbody);
        tableSection.appendChild(h('div', { class: 'top-units-wrap' }, table));
        container.appendChild(tableSection);
      }
    }
  },
};

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
